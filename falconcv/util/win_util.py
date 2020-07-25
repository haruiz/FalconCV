import os


class WinUtil:
    # wmi_conn = wmi.WMI() for local.
    @staticmethod
    def get_all_installed_win10_store():
        import wmi
        results = {}
        try:
            wmi_conn = wmi.WMI()
            for info in wmi_conn.Win32_InstalledStoreProgram():
                if '-' in info.Name: continue
                app_name = str(info.Name).encode('utf8', 'ignore').decode()
                if not info.Vendor:
                    vender = 'None'
                if '-' in info.Vendor:
                    vender = 'None'
                else:
                    vendor = str(info.Vendor).encode('utf8', 'ignore').decode()
                    vender = vendor.split(',')[0].split('=')[-1].strip()
                if app_name in results: continue
                if app_name not in results:
                    results[app_name] = []

                results[app_name] = [vender, info.Version, 'None']
        except:
            pass  # Handle errors here
        return results

    @staticmethod
    def get_installed_product_software():
        import wmi
        results = {}
        try:
            wmi_conn = wmi.WMI()
            for info in wmi_conn.Win32_Product():
                app_name = str(info.Name).encode('utf8', 'ignore').decode()
                vendor = str(info.Vendor).encode('utf8', 'ignore').decode()
                if app_name not in results:
                    results[app_name] = []
                results[app_name] = [vendor, info.Version, info.InstallDate]
        except:
            pass
        return results

    @staticmethod
    def get_all_installed_win10_software():
        import wmi
        results = {}
        try:
            wmi_conn = wmi.WMI()
            for info in wmi_conn.Win32_InstalledWin32Program():
                app_name = str(info.Name).encode('utf8', 'ignore').decode()
                vendor = str(info.Vendor).encode('utf8', 'ignore').decode()
                if app_name not in results:
                    results[app_name] = []
                results[app_name] = [vendor, info.Version, 'None']
        except:
            pass
        return results
